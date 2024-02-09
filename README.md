# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

### How to add a new doc
Everything you need to change for adding/deleting/modifying docs is in the `docs` folder. You don't have to touch anything else.

To add a doc just create a .md file in the docs directory. The directory structure also dictates the hierarchy. So for example creating a file in `docs/methods/dividemix.md`, means that it will be added to the methods (sub)section on the site.

All markdown syntax is valid, there's some additional stuff available (like danger/tip/info alerts) that you can check out [here](https://docusaurus.io/docs/markdown-features).

One more thing: don't forget about the frontmatter in your .md files (the properties between `---` at the top of the files), these set titles, you can use them to change the sidebar order etc. Read more [here](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-content-docs#markdown-front-matter).
### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```
$ USE_SSH=true yarn deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
